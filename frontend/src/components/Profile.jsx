import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import axios from 'axios';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const initialStats = {
  totalGestures: 0,
  accuracy: 0,
  favoriteGestures: [],
  recentActivity: [],
  progressData: {
    labels: [],
    datasets: [
      {
        label: 'Gesture Recognition Accuracy',
        data: [],
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1
      }
    ]
  }
};

const Profile = () => {
  const [userStats, setUserStats] = useState(initialStats);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchUserStats = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // In production, this would be an API call
        const response = await axios.get('http://127.0.0.1:5000/api/user/stats');
        setUserStats(response.data);
      } catch (error) {
        console.error('Error fetching user stats:', error);
        setError('Failed to load user statistics');
        
        // Fallback to mock data in development
        if (process.env.NODE_ENV === 'development') {
          const mockData = {
            totalGestures: 150,
            accuracy: 85,
            favoriteGestures: [
              { name: 'Hello', count: 25, emoji: '👋' },
              { name: 'Thank You', count: 18, emoji: '🙏' },
              { name: 'Yes', count: 15, emoji: '👍' }
            ],
            recentActivity: [
              { date: new Date().toISOString().split('T')[0], gesture: 'Hello', accuracy: 92 },
              { 
                date: new Date(Date.now() - 86400000).toISOString().split('T')[0], 
                gesture: 'Thank You', 
                accuracy: 88 
              },
              { 
                date: new Date(Date.now() - 172800000).toISOString().split('T')[0], 
                gesture: 'Yes', 
                accuracy: 95 
              }
            ],
            progressData: {
              labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
              datasets: [
                {
                  label: 'Gesture Recognition Accuracy',
                  data: [65, 72, 78, 85],
                  borderColor: 'rgb(75, 192, 192)',
                  tension: 0.1
                }
              ]
            }
          };
          setUserStats(mockData);
        }
      } finally {
        setLoading(false);
      }
    };

    fetchUserStats();
  }, []);

  if (loading) {
    return (
      <div className="p-6 bg-indigo-700 rounded-lg flex items-center justify-center">
        <div className="text-white">Loading profile data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 bg-indigo-700 rounded-lg">
        <div className="text-red-300 bg-red-900 p-4 rounded-lg">
          {error}
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 bg-indigo-700 rounded-lg">
      <h2 className="text-2xl font-bold mb-6 text-white">Your Profile</h2>
      
      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <div className="bg-indigo-600 p-4 rounded-lg shadow-lg hover:bg-indigo-500 transition-colors">
          <h3 className="text-lg font-semibold text-white">Total Gestures</h3>
          <p className="text-3xl font-bold text-purple-300">{userStats.totalGestures}</p>
        </div>
        <div className="bg-indigo-600 p-4 rounded-lg shadow-lg hover:bg-indigo-500 transition-colors">
          <h3 className="text-lg font-semibold text-white">Average Accuracy</h3>
          <p className="text-3xl font-bold text-purple-300">{userStats.accuracy}%</p>
        </div>
        <div className="bg-indigo-600 p-4 rounded-lg shadow-lg hover:bg-indigo-500 transition-colors">
          <h3 className="text-lg font-semibold text-white">Favorite Gesture</h3>
          <p className="text-3xl font-bold text-purple-300">
            {userStats.favoriteGestures[0]?.emoji || '👋'}
          </p>
        </div>
      </div>
      
      {/* Progress Chart */}
      <div className="bg-indigo-600 p-4 rounded-lg mb-8 shadow-lg">
        <h3 className="text-lg font-semibold text-white mb-4">Progress Over Time</h3>
        <div className="h-64 relative">
          <Line 
            data={userStats.progressData}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              scales: {
                y: {
                  beginAtZero: true,
                  max: 100,
                  grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                  },
                  ticks: {
                    color: 'white',
                    callback: value => `${value}%`
                  }
                },
                x: {
                  grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                  },
                  ticks: {
                    color: 'white'
                  }
                }
              },
              plugins: {
                legend: {
                  labels: {
                    color: 'white'
                  }
                },
                tooltip: {
                  backgroundColor: 'rgba(0, 0, 0, 0.8)',
                  titleColor: 'white',
                  bodyColor: 'white'
                }
              }
            }}
          />
        </div>
      </div>
      
      {/* Favorite Gestures */}
      <div className="bg-indigo-600 p-4 rounded-lg mb-8 shadow-lg">
        <h3 className="text-lg font-semibold text-white mb-4">Favorite Gestures</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {userStats.favoriteGestures.map((gesture, index) => (
            <div 
              key={index} 
              className="bg-indigo-700 p-4 rounded-lg shadow hover:bg-indigo-600 transition-colors"
            >
              <div className="flex items-center justify-between">
                <span className="text-2xl">{gesture.emoji}</span>
                <span className="text-white font-semibold">{gesture.name}</span>
              </div>
              <p className="text-purple-300 mt-2">Used {gesture.count} times</p>
            </div>
          ))}
        </div>
      </div>
      
      {/* Recent Activity */}
      <div className="bg-indigo-600 p-4 rounded-lg shadow-lg">
        <h3 className="text-lg font-semibold text-white mb-4">Recent Activity</h3>
        <div className="space-y-2">
          {userStats.recentActivity.map((activity, index) => (
            <div 
              key={index} 
              className="bg-indigo-700 p-3 rounded-lg flex items-center justify-between hover:bg-indigo-600 transition-colors"
            >
              <div>
                <span className="text-white font-semibold">{activity.gesture}</span>
                <span className="text-purple-300 text-sm ml-2">{activity.date}</span>
              </div>
              <span className={`text-${activity.accuracy >= 90 ? 'green' : 'purple'}-300`}>
                {activity.accuracy}% accuracy
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Profile; 
# WetRoute

When driving or cycling over long distances it is difficult to predict delays or inconvenience caused by weather. To solve this problem, I propose to develop an application to predict the delays caused by inclement weather along a route. Using the Project-OSRM (Open Source Routing Machine) engine to predict routes and the Weather Underground API to predict weather, the program can currently determine the weather along your route at the time you will be there. Moving forward a neural network will be incorporated to analyze past routes and determine the impact of actual weather type on travel durations.

The first plot shows a route with forecasted weather for a trip starting in Syracuse, NY and ending in Lake Placid, NY. The trip starts out clear, but is projected to become progressively cloudier and end during a thunderstorm.

The second plot shows a sample analysis dashboard using synthetic data which would look at the differences in driving speed relative to the speed limit for different weather conditions.

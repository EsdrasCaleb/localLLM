package com.hf.sfm.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import javax.servlet.http.HttpSession;
import static org.mockito.ArgumentMatchers.anyString;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class DataSource_getSession_3_1_Test {

    @Mock
    private HttpSession session;

    @InjectMocks
    private DataSource dataSource;

    @Test
    public void testSessionSuccess() {
        // Arrange
        String sessionName = "testSessionName";
        String sessionValue = "testSessionValue";
        when(session.getAttribute(anyString())).thenReturn(sessionValue);
        // Act
        String result = dataSource.getSession(session, sessionName);
        // Assert
        assertEquals(sessionValue, result);
    }

    @Test
    public void testSessionNullSession() {
        // Arrange
        String sessionName = "testSessionName";
        // Act and Assert
        assertThrows(NullPointerException.class, () -> dataSource.getSession(null, sessionName));
    }

    @Test
    public void testSessionNullSessionName() {
        // Arrange
        HttpSession session = mock(HttpSession.class);
        // Act and Assert
        assertThrows(NullPointerException.class, () -> dataSource.getSession(session, null));
    }

    @Test
    public void testSessionException() {
        // Arrange
        HttpSession session = mock(HttpSession.class);
        when(session.getAttribute(anyString())).thenThrow(Exception.class);
        // Act and Assert
        assertThrows(Exception.class, () -> dataSource.getSession(session, "testSessionName"));
    }
}

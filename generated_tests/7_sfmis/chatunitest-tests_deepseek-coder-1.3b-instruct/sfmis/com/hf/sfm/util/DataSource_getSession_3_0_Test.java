package com.hf.sfm.util;

import javax.servlet.http.HttpSession;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class DataSource_getSession_3_0_Test {

    @Test
    public void testGetSession() {
        // Arrange
        HttpSession session = mock(HttpSession.class);
        DataSource dataSource = new DataSource();
        String sessionName = "sessionName";
        String expectedSession = "sessionValue";
        // Mocking the session attribute
        when(session.getAttribute(sessionName)).thenReturn(expectedSession);
        // Act
        String actualSession = dataSource.getSession(session, sessionName);
        // Assert
        assertEquals(expectedSession, actualSession);
    }
}

package com.hf.sfm.util;

import javax.servlet.http.HttpSession;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class DataSource_getSession_3_0_Test {

    private DataSource dataSource;

    private HttpSession session;

    @BeforeEach
    public void setUp() {
        dataSource = new DataSource();
        session = mock(HttpSession.class);
    }

    @Test
    public void testGetSession_AttributeExists() {
        String sessionName = "user";
        String expectedValue = "JohnDoe";
        when(session.getAttribute(sessionName)).thenReturn(expectedValue);
        String result = dataSource.getSession(session, sessionName);
        assertEquals(expectedValue, result);
    }

    @Test
    public void testGetSession_AttributeIsNull() {
        String sessionName = "nonExistentAttribute";
        when(session.getAttribute(sessionName)).thenReturn(null);
        String result = dataSource.getSession(session, sessionName);
        assertNull(result);
    }

    @Test
    public void testGetSession_AttributeIsEmptyString() {
        String sessionName = "emptyAttribute";
        String expectedValue = "";
        when(session.getAttribute(sessionName)).thenReturn(expectedValue);
        String result = dataSource.getSession(session, sessionName);
        assertEquals(expectedValue, result);
    }
}

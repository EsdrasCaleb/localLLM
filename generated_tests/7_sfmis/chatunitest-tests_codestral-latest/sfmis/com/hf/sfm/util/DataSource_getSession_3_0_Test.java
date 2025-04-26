package com.hf.sfm.util;

import javax.servlet.http.HttpSession;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class DataSource_getSession_3_0_Test {

    @Mock
    private HttpSession mockSession;

    @InjectMocks
    private DataSource dataSource;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testGetSession() {
        String sessionName = "testSession";
        String expectedValue = "testValue";
        when(mockSession.getAttribute(sessionName)).thenReturn(expectedValue);
        String result = dataSource.getSession(mockSession, sessionName);
        assertEquals(expectedValue, result);
        verify(mockSession, times(1)).getAttribute(sessionName);
    }

    @Test
    public void testGetSessionNullAttribute() {
        String sessionName = "testSession";
        when(mockSession.getAttribute(sessionName)).thenReturn(null);
        String result = dataSource.getSession(mockSession, sessionName);
        assertEquals("null", result);
        verify(mockSession, times(1)).getAttribute(sessionName);
    }
}

package com.hf.sfm.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import javax.servlet.http.HttpSession;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class DataSource_getSession_3_1_Test {

    @Mock
    private HttpSession mockHttpSession;

    @InjectMocks
    private DataSource dataSource;

    @Test
    public void testGetSession() {
        when(mockHttpSession.getAttribute("sessionName")).thenReturn("sessionValue");
        String sessionValue = dataSource.getSession(mockHttpSession, "sessionName");
        assertEquals("sessionValue", sessionValue);
        verify(mockHttpSession).getAttribute("sessionName");
    }
}

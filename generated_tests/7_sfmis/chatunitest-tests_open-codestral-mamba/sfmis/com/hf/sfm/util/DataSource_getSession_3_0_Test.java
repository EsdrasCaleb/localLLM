package com.hf.sfm.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import javax.servlet.http.HttpSession;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class DataSource_getSession_3_0_Test {

    @Mock
    private HttpSession mockSession;

    @InjectMocks
    private DataSource dataSource;

    @BeforeEach
    public void setUp() {
        when(mockSession.getAttribute("sessionName")).thenReturn("sessionValue");
    }

    @Test
    public void testGetSession() {
        String result = dataSource.getSession(mockSession, "sessionName");
        assertEquals("sessionValue", result);
    }
}

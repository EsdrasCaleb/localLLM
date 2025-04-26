package com.hf.sfm.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import javax.servlet.http.HttpSession;

public class DataSource_getSession_3_0_Test {

    @Test
    public void testGetSession() {
        DataSource dataSource = new DataSource();
        HttpSession session = mock(HttpSession.class);
        String sessionName = "testSession";
        String expectedSessionValue = "testSession";
        when(session.getAttribute(sessionName)).thenReturn(expectedSessionValue);
        String actualSessionValue = dataSource.getSession(session, sessionName);
        // Assert the actual session value
        Assertions.assertEquals(expectedSessionValue, actualSessionValue);
    }
}

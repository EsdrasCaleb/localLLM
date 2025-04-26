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
    void testGetSession() {
        // given
        HttpSession mockSession = Mockito.mock(HttpSession.class);
        DataSource dataSource = new DataSource();
        // when
        when(mockSession.getAttribute(Mockito.anyString())).thenReturn("testSessionValue");
        // then
        assertEquals("testSessionValue", dataSource.getSession(mockSession, "sessionName"));
    }
}

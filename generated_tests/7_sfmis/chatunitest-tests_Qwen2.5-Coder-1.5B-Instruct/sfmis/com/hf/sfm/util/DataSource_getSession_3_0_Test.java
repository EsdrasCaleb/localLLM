package com.hf.sfm.util;

import javax.servlet.http.HttpSession;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class DataSource_getSession_3_0_Test {

    private DataSource dataSource;

    private HttpSession mockHttpSession;

    @BeforeEach
    public void setUp() {
        dataSource = new DataSource();
        mockHttpSession = mock(HttpSession.class);
    }

    @Test
    public void testGetSessionAttributeExists() throws Exception {
        // Set up the mock behavior for getAttribute method
        when(mockHttpSession.getAttribute("sessionName")).thenReturn("attributeValue");
        // Call the method under test
        String result = dataSource.getSession(mockHttpSession, "sessionName");
        // Verify the result
        assertEquals("attributeValue", result);
        // Check that getAttribute was called once
        verify(mockHttpSession).getAttribute("sessionName");
    }

    @Test
    public void testGetSessionAttributeDoesNotExist() throws Exception {
        // Set up the mock behavior for getAttribute method
        when(mockHttpSession.getAttribute("sessionName")).thenReturn(null);
        // Call the method under test
        String result = dataSource.getSession(mockHttpSession, "sessionName");
        // Verify the result
        assertEquals("", result);
        // Check that getAttribute was called once
        verify(mockHttpSession).getAttribute("sessionName");
    }
}

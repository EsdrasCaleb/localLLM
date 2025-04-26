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
    private HttpSession httpSession;

    @InjectMocks
    private DataSource dataSource;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testGetSession_AttributeExists() {
        // Arrange
        String sessionName = "testAttribute";
        String expectedValue = "testValue";
        when(httpSession.getAttribute(sessionName)).thenReturn(expectedValue);
        // Act
        String result = dataSource.getSession(httpSession, sessionName);
        // Assert
        assertEquals(expectedValue, result);
    }

    @Test
    public void testGetSession_AttributeDoesNotExist() {
        // Arrange
        String sessionName = "nonExistentAttribute";
        when(httpSession.getAttribute(sessionName)).thenReturn(null);
        // Act
        String result = dataSource.getSession(httpSession, sessionName);
        // Assert
        assertEquals("null", result);
    }

    @Test
    public void testGetSession_AttributeIsNonString() {
        // Arrange
        String sessionName = "nonStringAttribute";
        Integer nonStringValue = 123;
        when(httpSession.getAttribute(sessionName)).thenReturn(nonStringValue);
        // Act
        String result = dataSource.getSession(httpSession, sessionName);
        // Assert
        assertEquals(nonStringValue.toString(), result);
    }
}

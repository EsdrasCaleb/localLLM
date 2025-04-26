package com.ib.client;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class AnyWrapperMsgGenerator_ioError_4_0_Test {

    @Mock
    private Exception mockException;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testIoError() throws Exception {
        // Arrange
        String expectedErrorMessage = "Mocked Error Message";
        AnyWrapperMsgGenerator anyWrapperMsgGenerator = new AnyWrapperMsgGenerator();
        // Use reflection to mock the private method 'error'
        Method errorMethod = AnyWrapperMsgGenerator.class.getDeclaredMethod("error", Exception.class);
        errorMethod.setAccessible(true);
        when((String) errorMethod.invoke(anyWrapperMsgGenerator, mockException)).thenReturn(expectedErrorMessage);
        // Act
        String actualErrorMessage = AnyWrapperMsgGenerator.ioError(mockException);
        // Assert
        assertEquals(expectedErrorMessage, actualErrorMessage);
    }
}

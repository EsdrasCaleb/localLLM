package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

public class EWrapperMsgGenerator_receiveFA_25_0_Test {

    @Mock
    private EClientSocket eClientSocketMock;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testReceiveFA() {
        // Arrange
        int faDataType = 1;
        String xml = "<data>example</data>";
        String expectedTypeName = "TypeName";
        String expectedMessage = EWrapperMsgGenerator.FINANCIAL_ADVISOR + " " + expectedTypeName + " " + xml;
        // Mock the static method call
        try {
            // Use reflection to mock the static method
            final java.lang.reflect.Method method = EClientSocket.class.getDeclaredMethod("faMsgTypeName", int.class);
            method.setAccessible(true);
            when(method.invoke(null, faDataType)).thenReturn(expectedTypeName);
        } catch (Exception e) {
            fail("Failed to mock the static method", e);
        }
        // Act
        String result = EWrapperMsgGenerator.receiveFA(faDataType, xml);
        // Assert
        assertEquals(expectedMessage, result);
    }
}

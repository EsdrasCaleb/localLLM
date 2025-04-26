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

public class EWrapperMsgGenerator_receiveFA_25_2_Test {

    private static EClientSocket eClientSocketMock;

    @BeforeAll
    public static void setUp() {
        eClientSocketMock = mock(EClientSocket.class);
    }

    @Test
    public void testReceiveFA_ValidInput() {
        // Arrange
        int faDataType = 1;
        String xml = "<data>Test</data>";
        // Assuming this is the expected output of faMsgTypeName(1)
        String expectedMessageType = "MessageType1";
        // Mocking the static method
        try {
            java.lang.reflect.Method method = EClientSocket.class.getDeclaredMethod("faMsgTypeName", int.class);
            method.setAccessible(true);
            when(method.invoke(null, faDataType)).thenReturn(expectedMessageType);
        } catch (Exception e) {
            e.printStackTrace();
        }
        // Act
        String result = EWrapperMsgGenerator.receiveFA(faDataType, xml);
        // Assert
        assertEquals("FA: MessageType1 <data>Test</data>", result);
    }

    @Test
    public void testReceiveFA_EmptyXml() {
        // Arrange
        int faDataType = 2;
        String xml = "";
        // Assuming this is the expected output of faMsgTypeName(2)
        String expectedMessageType = "MessageType2";
        // Mocking the static method
        try {
            java.lang.reflect.Method method = EClientSocket.class.getDeclaredMethod("faMsgTypeName", int.class);
            method.setAccessible(true);
            when(method.invoke(null, faDataType)).thenReturn(expectedMessageType);
        } catch (Exception e) {
            e.printStackTrace();
        }
        // Act
        String result = EWrapperMsgGenerator.receiveFA(faDataType, xml);
        // Assert
        assertEquals("FA: MessageType2 ", result);
    }

    @Test
    public void testReceiveFA_NegativeFaDataType() {
        // Arrange
        int faDataType = -1;
        String xml = "<data>Negative Test</data>";
        // Assuming this is the expected output of faMsgTypeName(-1)
        String expectedMessageType = "MessageTypeNegative";
        // Mocking the static method
        try {
            java.lang.reflect.Method method = EClientSocket.class.getDeclaredMethod("faMsgTypeName", int.class);
            method.setAccessible(true);
            when(method.invoke(null, faDataType)).thenReturn(expectedMessageType);
        } catch (Exception e) {
            e.printStackTrace();
        }
        // Act
        String result = EWrapperMsgGenerator.receiveFA(faDataType, xml);
        // Assert
        assertEquals("FA: MessageTypeNegative <data>Negative Test</data>", result);
    }
}

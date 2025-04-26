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

public class EWrapperMsgGenerator_updateAccountTime_11_0_Test {

    @Test
    public void testUpdateAccountTime_withValidTimestamp() {
        // Arrange
        String timeStamp = "2023-10-01T10:00:00Z";
        String expected = "updateAccountTime: " + timeStamp;
        // Act
        String result = EWrapperMsgGenerator.updateAccountTime(timeStamp);
        // Assert
        assertEquals(expected, result);
    }

    @Test
    public void testUpdateAccountTime_withEmptyTimestamp() {
        // Arrange
        String timeStamp = "";
        String expected = "updateAccountTime: " + timeStamp;
        // Act
        String result = EWrapperMsgGenerator.updateAccountTime(timeStamp);
        // Assert
        assertEquals(expected, result);
    }

    @Test
    public void testUpdateAccountTime_withNullTimestamp() {
        // Arrange
        String timeStamp = null;
        String expected = "updateAccountTime: null";
        // Act
        String result = EWrapperMsgGenerator.updateAccountTime(timeStamp);
        // Assert
        assertEquals(expected, result);
    }
}

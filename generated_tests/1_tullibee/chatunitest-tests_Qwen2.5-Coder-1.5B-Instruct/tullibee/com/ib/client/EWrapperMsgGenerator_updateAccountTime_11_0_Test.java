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
    public void testUpdateAccountTime() {
        // Arrange
        EWrapperMsgGenerator generator = new EWrapperMsgGenerator();
        // Act
        String result = generator.updateAccountTime("2023-10-05T14:30:00");
        // Assert
        assertEquals("updateAccountTime: 2023-10-05T14:30:00", result);
    }

    @Test
    public void testUpdateAccountTimeWithNullTimestamp() {
        // Arrange
        EWrapperMsgGenerator generator = new EWrapperMsgGenerator();
        // Act
        String result = generator.updateAccountTime(null);
        // Assert
        assertEquals("updateAccountTime: null", result);
    }
}

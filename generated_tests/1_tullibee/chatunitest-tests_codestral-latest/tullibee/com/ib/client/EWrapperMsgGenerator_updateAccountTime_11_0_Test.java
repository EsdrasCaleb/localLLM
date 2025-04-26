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
        String timeStamp = "2023-10-05T14:48:00Z";
        String expected = "updateAccountTime: 2023-10-05T14:48:00Z";
        // Act
        String result = EWrapperMsgGenerator.updateAccountTime(timeStamp);
        // Assert
        assertEquals(expected, result);
    }
}

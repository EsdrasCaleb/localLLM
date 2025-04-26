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

class EWrapperMsgGenerator_currentTime_31_0_Test {

    @Test
    void currentTime() {
        // Arrange
        // Example time in milliseconds
        long time = 1633456800L;
        String expectedTime = "current time = 1633456800 (" + DateFormat.getDateTimeInstance().format(new Date(time * 1000)) + ")";
        // Act
        String actualTime = EWrapperMsgGenerator.currentTime(time);
        // Assert
        assert actualTime.equals(expectedTime);
    }
}

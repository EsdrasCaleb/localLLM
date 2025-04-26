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

public class EWrapperMsgGenerator_updateAccountTime_11_1_Test {

    @Test
    public void testUpdateAccountTime() {
        // Arrange
        EWrapperMsgGenerator eWrapperMsgGenerator = new EWrapperMsgGenerator();
        String timeStamp = "2020-01-01 12:00:00";
        // Act
        String result = eWrapperMsgGenerator.updateAccountTime(timeStamp);
        // Assert
        assertEquals("updateAccountTime: 2020-01-01 12:00:00", result);
    }
}

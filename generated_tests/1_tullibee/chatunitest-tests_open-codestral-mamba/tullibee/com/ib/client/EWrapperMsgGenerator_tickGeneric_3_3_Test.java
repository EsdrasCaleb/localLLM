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

public class EWrapperMsgGenerator_tickGeneric_3_3_Test {

    @Test
    public void testTickGeneric() {
        int testTickerId = 1;
        int testTickType = 2;
        double testValue = 3.14;
        String expectedResult = "id=" + testTickerId + "  " + TickType.getField(testTickType) + "=" + testValue;
        String actualResult = EWrapperMsgGenerator.tickGeneric(testTickerId, testTickType, testValue);
        assertEquals(expectedResult, actualResult);
    }
}

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

public class EWrapperMsgGenerator_tickOptionComputation_2_0_Test {

    @Test
    public void testTickOptionComputation() {
        int tickerId = 123;
        int field = 4;
        double impliedVol = 0.123;
        double delta = 0.456;
        double modelPrice = 7.89;
        double pvDividend = 0.01;
        String expectedResult = "id=123  TickType.getField(4): vol = 0.123 delta = 0.456: modelPrice = 7.89: pvDividend = 0.01";
        String actualResult = EWrapperMsgGenerator.tickOptionComputation(tickerId, field, impliedVol, delta, modelPrice, pvDividend);
        assertEquals(expectedResult, actualResult);
    }
}

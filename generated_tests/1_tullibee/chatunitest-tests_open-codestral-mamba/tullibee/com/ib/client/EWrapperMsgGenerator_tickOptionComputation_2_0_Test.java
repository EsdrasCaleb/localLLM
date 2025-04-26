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
        int tickerId = 1;
        int field = TickType.MODEL_OPTION;
        double impliedVol = 0.2;
        double delta = 0.5;
        double modelPrice = 100.0;
        double pvDividend = 5.0;
        String expectedResult = "id=1  OPTION : vol = 0.2 delta = 0.5 : modelPrice = 100.0 : pvDividend = 5.0";
        String result = EWrapperMsgGenerator.tickOptionComputation(tickerId, field, impliedVol, delta, modelPrice, pvDividend);
        assertEquals(expectedResult, result);
    }
}

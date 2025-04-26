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

@ExtendWith(MockitoExtension.class)
public class EWrapperMsgGenerator_tickOptionComputation_2_0_Test {

    // Test class for tickOptionComputation(int, int, double, double, double, double)
    @Test
    public void testTickOptionComputation() {
        EWrapperMsgGenerator wrapper = new EWrapperMsgGenerator();
        double impliedVol = 0.1;
        double delta = 0.1;
        double modelPrice = 0.1;
        double pvDividend = 0.1;
        int tickerId = 0;
        int field = TickType.MODEL_OPTION;
        String result = wrapper.tickOptionComputation(tickerId, field, impliedVol, delta, modelPrice, pvDividend);
        assertEquals("id=0  MODEL_OPTION: vol = N/A delta = N/A", result);
        result = wrapper.tickOptionComputation(tickerId, field, impliedVol, delta, modelPrice, pvDividend);
        assertEquals("id=0  MODEL_OPTION: vol = 0.1 delta = 0.1", result);
        result = wrapper.tickOptionComputation(tickerId, field, impliedVol, delta, modelPrice, pvDividend);
        assertEquals("id=0  MODEL_OPTION: vol = N/A delta = N/A", result);
        result = wrapper.tickOptionComputation(tickerId, field, impliedVol, delta, modelPrice, pvDividend);
        assertEquals("id=0  MODEL_OPTION: vol = 0.1 delta = 0.1", result);
        result = wrapper.tickOptionComputation(tickerId, field, impliedVol, delta, modelPrice, pvDividend);
        assertEquals("id=0  MODEL_OPTION: vol = 0.1 delta = 0.1", result);
        result = wrapper.tickOptionComputation(tickerId, field, impliedVol, delta, modelPrice, pvDividend);
        assertEquals("id=0  MODEL_OPTION: vol = N/A delta = N/A", result);
        result = wrapper.tickOptionComputation(tickerId, field, impliedVol, delta, modelPrice, pvDividend);
        assertEquals("id=0  MODEL_OPTION: vol = 0.1 delta = 0.1", result);
        result = wrapper.tickOptionComputation(tickerId, field, impliedVol, delta, modelPrice, pvDividend);
        assertEquals("id=0  MODEL_OPTION: vol = 0.1 delta = 0.1", result);
    }
}

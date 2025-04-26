package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

class EWrapperMsgGenerator_tickOptionComputation_2_0_Test {

    @Test
    void testTickOptionComputationAllValid() {
        String result = EWrapperMsgGenerator.tickOptionComputation(1, TickType.MODEL_OPTION, 0.25, 0.5, 100.0, 5.0);
        assertEquals("id=1  MODEL_OPTION: vol = 0.25 delta = 0.5: modelPrice = 100.0: pvDividend = 5.0", result);
    }

    @Test
    void testTickOptionComputationInvalidVol() {
        String result = EWrapperMsgGenerator.tickOptionComputation(1, TickType.MODEL_OPTION, -0.25, 0.5, 100.0, 5.0);
        assertEquals("id=1  MODEL_OPTION: vol = N/A delta = 0.5: modelPrice = 100.0: pvDividend = 5.0", result);
    }

    @Test
    void testTickOptionComputationInvalidDelta() {
        String result = EWrapperMsgGenerator.tickOptionComputation(1, TickType.MODEL_OPTION, 0.25, 1.5, 100.0, 5.0);
        assertEquals("id=1  MODEL_OPTION: vol = 0.25 delta = N/A: modelPrice = 100.0: pvDividend = 5.0", result);
    }

    @Test
    void testTickOptionComputationInvalidModelPrice() {
        String result = EWrapperMsgGenerator.tickOptionComputation(1, TickType.MODEL_OPTION, 0.25, 0.5, -100.0, 5.0);
        assertEquals("id=1  MODEL_OPTION: vol = 0.25 delta = 0.5: modelPrice = N/A: pvDividend = 5.0", result);
    }

    @Test
    void testTickOptionComputationInvalidPvDividend() {
        String result = EWrapperMsgGenerator.tickOptionComputation(1, TickType.MODEL_OPTION, 0.25, 0.5, 100.0, -5.0);
        assertEquals("id=1  MODEL_OPTION: vol = 0.25 delta = 0.5: modelPrice = 100.0: pvDividend = N/A", result);
    }

    @Test
    void testTickOptionComputationMaxVol() {
        String result = EWrapperMsgGenerator.tickOptionComputation(1, TickType.MODEL_OPTION, Double.MAX_VALUE, 0.5, 100.0, 5.0);
        assertEquals("id=1  MODEL_OPTION: vol = N/A delta = 0.5: modelPrice = 100.0: pvDividend = 5.0", result);
    }

    @Test
    void testTickOptionComputationMaxModelPrice() {
        String result = EWrapperMsgGenerator.tickOptionComputation(1, TickType.MODEL_OPTION, 0.25, 0.5, Double.MAX_VALUE, 5.0);
        assertEquals("id=1  MODEL_OPTION: vol = 0.25 delta = 0.5: modelPrice = N/A: pvDividend = 5.0", result);
    }

    @Test
    void testTickOptionComputationMaxPvDividend() {
        String result = EWrapperMsgGenerator.tickOptionComputation(1, TickType.MODEL_OPTION, 0.25, 0.5, 100.0, Double.MAX_VALUE);
        assertEquals("id=1  MODEL_OPTION: vol = 0.25 delta = 0.5: modelPrice = 100.0: pvDividend = N/A", result);
    }

    @Test
    void testTickOptionComputationOtherField() {
        String result = EWrapperMsgGenerator.tickOptionComputation(1, TickType.BID_SIZE, 0.25, 0.5, 100.0, 5.0);
        assertEquals("id=1  BID_SIZE: vol = 0.25 delta = 0.5", result);
    }
}

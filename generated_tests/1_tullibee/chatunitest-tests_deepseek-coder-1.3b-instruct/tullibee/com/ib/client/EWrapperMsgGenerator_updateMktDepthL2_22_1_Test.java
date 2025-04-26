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

public class EWrapperMsgGenerator_updateMktDepthL2_22_1_Test {

    @Test
    public void testUpdateMktDepthL2() {
        // Test with all valid parameters
        String result = EWrapperMsgGenerator.updateMktDepthL2(1, 2, "NASDAQ", 1, 1, 100.0, 100);
        assertEquals("updateMktDepth: 1 2 NASDAQ 1 1 100.0 100", result);
        // Test with invalid parameters (negative size)
        result = EWrapperMsgGenerator.updateMktDepthL2(1, 2, "NASDAQ", 1, 1, -100.0, 100);
        assertEquals("updateMktDepth: 1 2 NASDAQ 1 1 -100.0 100", result);
        // Test with invalid parameters (negative price)
        result = EWrapperMsgGenerator.updateMktDepthL2(1, 2, "NASDAQ", 1, 1, 100.0, -100);
        assertEquals("updateMktDepth: 1 2 NASDAQ 1 1 100.0 -100", result);
        // Test with invalid parameters (negative position)
        result = EWrapperMsgGenerator.updateMktDepthL2(1, -2, "NASDAQ", 1, 1, 100.0, 100);
        assertEquals("updateMktDepth: 1 -2 NASDAQ 1 1 100.0 100", result);
        // Test with invalid parameters (negative tickerId)
        result = EWrapperMsgGenerator.updateMktDepthL2(-1, 2, "NASDAQ", 1, 1, 100.0, 100);
        assertEquals("updateMktDepth: -1 2 NASDAQ 1 1 100.0 100", result);
        // Test with invalid parameters (negative operation)
        result = EWrapperMsgGenerator.updateMktDepthL2(1, 2, "NASDAQ", -1, 1, 100.0, 100);
        assertEquals("updateMktDepth: 1 2 NASDAQ -1 1 100.0 100", result);
        // Test with invalid parameters (negative side)
        result = EWrapperMsgGenerator.updateMktDepthL2(1, 2, "NASDAQ", 1, -1, 100.0, 100);
        assertEquals("updateMktDepth: 1 2 NASDAQ 1 -1 100.0 100", result);
    }
}

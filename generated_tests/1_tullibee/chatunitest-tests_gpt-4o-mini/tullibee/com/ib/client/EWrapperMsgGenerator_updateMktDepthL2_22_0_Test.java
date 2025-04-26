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

public class EWrapperMsgGenerator_updateMktDepthL2_22_0_Test {

    @Test
    public void testUpdateMktDepthL2() {
        // Test case 1: Normal case with valid inputs
        String result1 = EWrapperMsgGenerator.updateMktDepthL2(1, 0, "MarketMakerA", 0, 1, 100.50, 10);
        assertEquals("updateMktDepth: 1 0 MarketMakerA 0 1 100.5 10", result1);
        // Test case 2: Different market maker
        String result2 = EWrapperMsgGenerator.updateMktDepthL2(2, 1, "MarketMakerB", 1, 0, 200.75, 20);
        assertEquals("updateMktDepth: 2 1 MarketMakerB 1 0 200.75 20", result2);
        // Test case 3: Edge case with zero size
        String result3 = EWrapperMsgGenerator.updateMktDepthL2(3, 2, "MarketMakerC", 0, 1, 150.00, 0);
        assertEquals("updateMktDepth: 3 2 MarketMakerC 0 1 150.0 0", result3);
        // Test case 4: Negative values for position and size
        String result4 = EWrapperMsgGenerator.updateMktDepthL2(4, -1, "MarketMakerD", 1, -1, 50.25, -5);
        assertEquals("updateMktDepth: 4 -1 MarketMakerD 1 -1 50.25 -5", result4);
        // Test case 5: Large values for price and size
        String result5 = EWrapperMsgGenerator.updateMktDepthL2(5, 3, "MarketMakerE", 0, 1, 1000000.99, 10000);
        assertEquals("updateMktDepth: 5 3 MarketMakerE 0 1 1000000.99 10000", result5);
    }
}

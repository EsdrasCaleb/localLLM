package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_numberOfRows_21_0_Test {

    @Test
    void testNumberOfRows() {
        ScannerSubscription subscription = new ScannerSubscription();
        // Test with a valid number of rows
        int validNum = 10;
        subscription.numberOfRows(validNum);
        assertEquals(validNum, subscription.numberOfRows());
        // Test with zero rows
        subscription.numberOfRows(0);
        assertEquals(0, subscription.numberOfRows());
        // Test with negative rows
        int negativeNum = -5;
        subscription.numberOfRows(negativeNum);
        assertEquals(negativeNum, subscription.numberOfRows());
        // Test with the maximum integer value
        int maxInt = Integer.MAX_VALUE;
        subscription.numberOfRows(maxInt);
        assertEquals(maxInt, subscription.numberOfRows());
        // Test with the minimum integer value
        int minInt = Integer.MIN_VALUE;
        subscription.numberOfRows(minInt);
        assertEquals(minInt, subscription.numberOfRows());
    }
}

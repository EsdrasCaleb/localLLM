package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_numberOfRows_21_0_Test {

    @Test
    void testNumberOfRows_validInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        int expectedRows = 10;
        subscription.numberOfRows(expectedRows);
        assertEquals(expectedRows, subscription.numberOfRows());
    }

    @Test
    void testNumberOfRows_negativeInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        int expectedRows = -5;
        subscription.numberOfRows(expectedRows);
        assertEquals(expectedRows, subscription.numberOfRows());
    }

    @Test
    void testNumberOfRows_zeroInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        int expectedRows = 0;
        subscription.numberOfRows(expectedRows);
        assertEquals(expectedRows, subscription.numberOfRows());
    }

    @Test
    void testNumberOfRows_largeInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        int expectedRows = 100000;
        subscription.numberOfRows(expectedRows);
        assertEquals(expectedRows, subscription.numberOfRows());
    }

    @Test
    void testNumberOfRows_default() {
        ScannerSubscription subscription = new ScannerSubscription();
        assertEquals(ScannerSubscription.NO_ROW_NUMBER_SPECIFIED, subscription.numberOfRows());
    }
}

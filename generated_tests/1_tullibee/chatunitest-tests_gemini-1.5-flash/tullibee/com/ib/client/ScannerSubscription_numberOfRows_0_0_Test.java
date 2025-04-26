package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_numberOfRows_0_0_Test {

    @Test
    void testNumberOfRows_noRowsSpecified() {
        ScannerSubscription subscription = new ScannerSubscription();
        assertEquals(ScannerSubscription.NO_ROW_NUMBER_SPECIFIED, subscription.numberOfRows());
    }

    @Test
    void testNumberOfRows_rowsSpecified() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.numberOfRows(10);
        assertEquals(10, subscription.numberOfRows());
    }

    @Test
    void testNumberOfRows_zeroRowsSpecified() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.numberOfRows(0);
        assertEquals(0, subscription.numberOfRows());
    }

    @Test
    void testNumberOfRows_negativeRowsSpecified() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.numberOfRows(-5);
        assertEquals(-5, subscription.numberOfRows());
    }
}

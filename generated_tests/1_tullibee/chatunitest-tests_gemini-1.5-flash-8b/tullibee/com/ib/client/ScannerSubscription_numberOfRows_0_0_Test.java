package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_numberOfRows_0_0_Test {

    @Test
    void testNumberOfRows_noRowsSpecified() {
        ScannerSubscription subscription = new ScannerSubscription();
        int rows = subscription.numberOfRows();
        assertEquals(ScannerSubscription.NO_ROW_NUMBER_SPECIFIED, rows);
    }

    @Test
    void testNumberOfRows_rowsSpecified() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.numberOfRows(10);
        int rows = subscription.numberOfRows();
        assertEquals(10, rows);
    }
}

package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_numberOfRows_21_0_Test {

    @Test
    public void testNumberOfRows() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.numberOfRows(5);
        assertEquals(5, scannerSubscription.numberOfRows());
    }

    @Test
    public void testNumberOfRowsSetToNegativeNumber() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.numberOfRows(-1);
        assertEquals(-1, scannerSubscription.numberOfRows());
    }

    @Test
    public void testNumberOfRowsSetToZero() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.numberOfRows(0);
        assertEquals(0, scannerSubscription.numberOfRows());
    }

    @Test
    public void testNumberOfRowsSetToPositiveNumber() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.numberOfRows(10);
        assertEquals(10, scannerSubscription.numberOfRows());
    }

    @Test
    public void testNumberOfRowsSetToLargeNumber() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.numberOfRows(Integer.MAX_VALUE);
        assertEquals(Integer.MAX_VALUE, scannerSubscription.numberOfRows());
    }
}

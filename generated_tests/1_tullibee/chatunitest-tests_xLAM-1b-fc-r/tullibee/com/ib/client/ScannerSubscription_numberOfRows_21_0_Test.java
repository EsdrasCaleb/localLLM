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
        ScannerSubscription subscription = new ScannerSubscription();
        int expectedNumberOfRows = 10;
        subscription.numberOfRows(expectedNumberOfRows);
        assertEquals(expectedNumberOfRows, subscription.numberOfRows());
        subscription.numberOfRows(-1);
        assertEquals(ScannerSubscription.NO_ROW_NUMBER_SPECIFIED, subscription.numberOfRows());
        subscription.numberOfRows(0);
        assertEquals(0, subscription.numberOfRows());
    }
}

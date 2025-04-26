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
        int numRows = 10;
        subscription.numberOfRows(numRows);
        assertEquals(numRows, subscription.numberOfRows());
    }
}

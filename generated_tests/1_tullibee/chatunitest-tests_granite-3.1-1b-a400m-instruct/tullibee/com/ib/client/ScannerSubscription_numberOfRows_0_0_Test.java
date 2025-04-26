package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_numberOfRows_0_0_Test {

    @Test
    public void testNumberOfRows() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.numberOfRows(5);
        assertEquals(5, scannerSubscription.numberOfRows());
    }
}

package com.ib.client;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_numberOfRows_21_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @ParameterizedTest
    @ValueSource(ints = { 0, 1, 10, 100, Integer.MAX_VALUE })
    public void testNumberOfRows(int num) {
        scannerSubscription.numberOfRows(num);
        assertEquals(num, scannerSubscription.numberOfRows());
    }

    @Test
    public void testNumberOfRowsNegative() {
        scannerSubscription.numberOfRows(-1);
        assertEquals(-1, scannerSubscription.numberOfRows());
    }

    @Test
    public void testNumberOfRowsDefault() {
        assertEquals(ScannerSubscription.NO_ROW_NUMBER_SPECIFIED, scannerSubscription.numberOfRows());
    }
}

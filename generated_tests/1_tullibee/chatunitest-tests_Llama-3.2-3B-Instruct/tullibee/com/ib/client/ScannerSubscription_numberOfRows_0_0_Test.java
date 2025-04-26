package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_numberOfRows_0_0_Test {

    @Test
    public void testNumberOfRows_ReturnsNO_ROW_NUMBER_SPECIFIED() {
        ScannerSubscription subscription = new ScannerSubscription();
        assertEquals(ScannerSubscription.NO_ROW_NUMBER_SPECIFIED, subscription.numberOfRows());
    }

    @Test
    public void testNumberOfRows_ReturnsCorrectValue() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.numberOfRows(10);
        assertEquals(10, subscription.numberOfRows());
    }

    @Test
    public void testNumberOfRows_ReturnsCorrectValueWithMultipleInvocations() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.numberOfRows(10);
        subscription.numberOfRows(20);
        assertEquals(20, subscription.numberOfRows());
    }

    @Test
    public void testNumberOfRows_ThrowsNullPointerException_WhenNullSubscription() {
        ScannerSubscription subscription = null;
        assertThrows(NullPointerException.class, () -> subscription.numberOfRows());
    }
}

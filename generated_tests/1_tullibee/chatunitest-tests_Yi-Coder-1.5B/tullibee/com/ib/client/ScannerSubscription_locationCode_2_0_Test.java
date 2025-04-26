package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
public class ScannerSubscription_locationCode_2_0_Test {

    @Test
    public void testLocationCode() {
        ScannerSubscription s = new ScannerSubscription();
        s.locationCode("123456");
        assertEquals("123456", s.locationCode());
    }
}

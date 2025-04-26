package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
public class ScannerSubscription_aboveVolume_6_2_Test {

    // Test class
    @Test
    public void test_aboveVolume() {
        ScannerSubscription s = new ScannerSubscription();
        s.aboveVolume(20);
        assert s.aboveVolume() == 20;
    }
}

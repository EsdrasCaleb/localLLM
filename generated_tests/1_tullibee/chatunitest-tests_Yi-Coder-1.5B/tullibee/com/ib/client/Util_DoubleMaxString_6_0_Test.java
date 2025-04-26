package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class Util_DoubleMaxString_6_0_Test {

    @Test
    public void TestDoubleMaxString() {
        // arrange
        double value = 10.0;
        // act
        String actual = Util.DoubleMaxString(value);
        // assert
        Assertions.assertEquals("10.0", actual);
    }
}

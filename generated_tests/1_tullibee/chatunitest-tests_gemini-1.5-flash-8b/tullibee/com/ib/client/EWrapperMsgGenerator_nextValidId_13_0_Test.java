package com.ib.client;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

public class EWrapperMsgGenerator_nextValidId_13_0_Test {

    @ParameterizedTest
    @CsvSource({ "1, Next Valid Order ID: 1", "10, Next Valid Order ID: 10", "0, Next Valid Order ID: 0" })
    void testNextValidId(int orderId, String expected) {
        String actual = EWrapperMsgGenerator.nextValidId(orderId);
        assertEquals(expected, actual);
    }
}

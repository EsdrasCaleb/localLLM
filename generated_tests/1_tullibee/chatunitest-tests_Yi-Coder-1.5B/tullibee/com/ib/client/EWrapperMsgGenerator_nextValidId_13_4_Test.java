package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

class EWrapperMsgGenerator_nextValidId_13_4_Test {

    @Test
    void test1() {
        EWrapperMsgGenerator wrapper = new EWrapperMsgGenerator();
        assertEquals("Next Valid Order ID: 123", wrapper.nextValidId(123));
    }
}

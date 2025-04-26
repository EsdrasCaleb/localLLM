package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
public class TickType_getField_0_2_Test {

    @Test
    public void testGetField() {
        // <Buggy Line>: reference to assertEquals is ambiguous  both method assertEquals(java.lang.Object,java.lang.Object) in org.junit.Assert and method assertEquals(java.lang.Object,java.lang.Object) in org.junit.jupiter.api.Assertions match
        assertEquals("bidSize", TickType.getField(TickType.BID_SIZE));
        // <Buggy Line>: reference to assertEquals is ambiguous  both method assertEquals(java.lang.Object,java.lang.Object) in org.junit.Assert and method assertEquals(java.lang.Object,java.lang.Object) in org.junit.jupiter.api.Assertions match
        assertEquals("bidPrice", TickType.getField(TickType.BID));
        // <Buggy Line>: reference to assertEquals is ambiguous  both method assertEquals(java.lang.Object,java.lang.Object) in org.junit.Assert and method assertEquals(java.lang.Object,java.lang.Object) in org.junit.jupiter.api.Assertions match
        assertEquals("askPrice", TickType.getField(TickType.ASK));
        // <Buggy Line>: reference to assertEquals is ambiguous  both method assertEquals(java.lang.Object,java.lang.Object) in org.junit.Assert and method assertEquals(java.lang.Object,java.lang.Object) in org.junit.jupiter.api.Assertions match
        assertEquals("askSize", TickType.getField(TickType.ASK_SIZE));
        // <Buggy Line>: reference to assertEquals is ambiguous  both method assertEquals(java.lang.Object,java.lang.Object) in org.junit.Assert and method assertEquals(java.lang.Object,java.lang.Object) in org.junit.jupiter.api.Assertions match
        assertEquals("lastPrice", TickType.getField(TickType.LAST));
        // <Buggy Line>: reference to assertEquals is ambiguous  both method assertEquals(java.lang.Object,java.lang.Object) in org.junit.Assert and method assertEquals(java.lang.Object,java.lang.Object) in org.junit.jupiter.api.Assertions match
        assertEquals("lastSize", TickType.getField(TickType.LAST_SIZE));
        // <Buggy Line>: reference to assertEquals is ambiguous  both method assertEquals(java.lang.Object,java.lang.Object) in org.junit.Assert and method assertEquals(java.lang.Object,java.lang.Object) in org.junit.jupiter.api.Assertions match
        assertEquals("high", TickType.getField(TickType.HIGH));
        // <Buggy Line>: reference to assertEquals is ambiguous  both method assertEquals(java.lang.Object,java.lang.Object) in org.junit.Assert and method assertEquals(java.lang.Object,java.lang.Object) in org.junit.jupiter.api.Assertions match
        assertEquals("low", TickType.getField(TickType.LOW));
        // <Buggy Line>: reference to assertEquals is ambiguous  both method assertEquals(java.lang.Object,java.lang.Object) in org.junit.Assert and method assertEquals(java.lang.Object,java.lang.Object) in org.junit.jupiter.api.Assertions match
        assertEquals("volume", TickType.getField(TickType.VOLUME));
        // <Buggy Line>: reference to assertEquals is ambiguous  both method assertEquals(java.lang.Object,java.lang.Object) in org.junit.Assert and method assertEquals(java.lang.Object,java.lang.Object) in org.junit.jupiter.api.Assertions match
        assertEquals("close", TickType.getField(TickType.CLOSE));
        // <Buggy Line>: reference to assertEquals is ambiguous  both method assertEquals(java.lang.Object,java.lang.Object) in org.junit.Assert and method assertEquals(java.lang.Object,java.lang.Object) in org.junit.jupiter.api.Assertions match
        assertEquals("bidOptComp", TickType.getField(TickType.BID_OPTION));
    }
}

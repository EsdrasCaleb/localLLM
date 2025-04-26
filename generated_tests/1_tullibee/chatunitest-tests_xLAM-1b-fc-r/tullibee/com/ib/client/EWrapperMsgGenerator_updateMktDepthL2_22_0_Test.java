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

public class EWrapperMsgGenerator_updateMktDepthL2_22_0_Test {

    @Test
    public void testUpdateMktDepthL2() {
        int tickerId = 123;
        int position = 456;
        String marketMaker = "XYZ";
        int operation = 789;
        int side = 123;
        double price = 45.67;
        int size = 890;
        String expected = "updateMktDepth: 123 456 XYZ 789 123 45.67 890";
        String actual = EWrapperMsgGenerator.updateMktDepthL2(tickerId, position, marketMaker, operation, side, price, size);
        assertEquals(expected, actual);
    }
}

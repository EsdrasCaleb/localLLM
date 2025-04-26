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
        int tickerId = 1;
        int position = 2;
        String marketMaker = "MM1";
        int operation = 1;
        int side = 1;
        double price = 100.50;
        int size = 10;
        String expected = "updateMktDepth: 1 2 MM1 1 1 100.5 10";
        String result = EWrapperMsgGenerator.updateMktDepthL2(tickerId, position, marketMaker, operation, side, price, size);
        assertEquals(expected, result);
    }
}

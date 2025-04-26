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

public class EWrapperMsgGenerator_updateMktDepth_21_0_Test {

    @InjectMocks
    private EWrapperMsgGenerator eWrapperMsgGenerator;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testUpdateMktDepth() {
        int tickerId = 1;
        int position = 2;
        int operation = 3;
        int side = 4;
        double price = 100.50;
        int size = 10;
        String expected = "updateMktDepth: 1 2 3 4 100.5 10";
        String result = eWrapperMsgGenerator.updateMktDepth(tickerId, position, operation, side, price, size);
        assertEquals(expected, result);
    }
}

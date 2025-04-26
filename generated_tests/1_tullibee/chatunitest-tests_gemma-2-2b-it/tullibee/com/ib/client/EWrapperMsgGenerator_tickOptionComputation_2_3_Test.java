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

public class EWrapperMsgGenerator_tickOptionComputation_2_3_Test {

    @Test
    void tickOptionComputation() {
        EWrapperMsgGenerator.tickOptionComputation(1, TickType.MODEL_OPTION, 1.0, 0.2, 10.0, 0.0);
    }
}

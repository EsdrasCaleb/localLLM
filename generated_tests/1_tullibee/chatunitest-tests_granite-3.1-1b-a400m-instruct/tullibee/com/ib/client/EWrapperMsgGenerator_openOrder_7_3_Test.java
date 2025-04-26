package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class EWrapperMsgGenerator_openOrder_7_3_Test {

    private EWrapperMsgGenerator msgGenerator;

    @BeforeEach
    public void setUp() {
        msgGenerator = new EWrapperMsgGenerator();
    }
}

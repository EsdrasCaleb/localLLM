package com.densebrain.rif.client;

import java.rmi.RemoteException;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Hashtable;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;

class RIFManagerFactory_getManager_1_1_Test {

    @Mock
    private RIFManager mockRIFManager;

    @InjectMocks
    private RIFManagerFactory rifManagerFactory;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }
}

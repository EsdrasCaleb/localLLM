package com.densebrain.rif.client;

import java.rmi.RemoteException;
import java.util.Hashtable;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class RIFManagerFactory_getInvoker_2_4_Test {

    @Mock
    private RIFManager mockManager;

    @Mock
    private RIFInvoker mockInvoker;

    @InjectMocks
    private RIFManagerFactory factory;

    @Test
    void testGetInvoker_ValidInput_ReturnsInvoker() throws RemoteException {
        Hashtable<String, RIFManager> managerMap = new Hashtable<>();
        managerMap.put("testUrl", mockManager);
        // Using Mockito's @InjectMocks annotation to instantiate the class under test
        when(mockManager.getInvoker(any())).thenReturn(mockInvoker);
        RIFInvoker invoker = factory.getInvoker("testUrl", String.class);
        assertEquals(mockInvoker, invoker);
        verify(mockManager).getInvoker(String.class);
    }

    @Test
    void testGetInvoker_InvalidUrl_ThrowsRemoteException() {
        assertThrows(RemoteException.class, () -> factory.getInvoker("invalidUrl", String.class));
    }

    @Test
    void testGetInvoker_NullUrl_ThrowsRemoteException() {
        assertThrows(RemoteException.class, () -> factory.getInvoker(null, String.class));
    }

    @Test
    void testGetInvoker_NullInterface_ThrowsRemoteException() {
        Hashtable<String, RIFManager> managerMap = new Hashtable<>();
        managerMap.put("testUrl", mockManager);
        // Using Mockito's @InjectMocks annotation to instantiate the class under test
        assertThrows(RemoteException.class, () -> factory.getInvoker("testUrl", null));
    }
}

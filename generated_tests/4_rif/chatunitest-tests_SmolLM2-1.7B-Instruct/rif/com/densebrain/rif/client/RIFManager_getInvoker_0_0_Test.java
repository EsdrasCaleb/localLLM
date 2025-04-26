package com.densebrain.rif.client;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;
import java.util.Hashtable;
import java.util.Map;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.rmi.RemoteException;
import com.densebrain.rif.client.service.RIFService;
import com.densebrain.rif.client.service.RIFServiceStub;

@RunWith(MockitoJUnitRunner.class)
public class RIFManager_getInvoker_0_0_Test {

    @Mock
    private RIFService service;

    @Mock
    private RIFClassLoader classLoader;

    @InjectMocks
    private RIFManager rifManager;

    private RIFInvoker invoker;

    private Class interfaceClazz;

    @Test
    public void testGetInvoker_InterfaceClassNotPresent_InvokerCreated() throws RemoteException {
        interfaceClazz = RIFInvoker.class;
        when(rifManager.getInvoker(interfaceClazz)).thenReturn(invoker);
        assertEquals(invoker, rifManager.getInvoker(interfaceClazz));
    }

    @Test
    public void testGetInvoker_InterfaceClassPresent_InvokerRetrieved() throws RemoteException {
        interfaceClazz = RIFInvoker.class;
        when(rifManager.getInvoker(interfaceClazz)).thenReturn(invoker);
        assertEquals(invoker, rifManager.getInvoker(interfaceClazz));
    }

    @Test
    public void testGetInvoker_InterfaceClassNotPresent_InvokerNotCreated() throws RemoteException {
        interfaceClazz = RIFInvoker.class;
        when(rifManager.getInvoker(interfaceClazz)).thenReturn(null);
        assertNotNull(rifManager.getInvoker(interfaceClazz));
    }
}

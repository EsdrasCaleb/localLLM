package com.densebrain.rif.client;

import java.rmi.RemoteException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Hashtable;

public class RIFManagerFactory_getManager_1_3_Test {

    @Test
    public void testGetManager() throws RemoteException {
        // given
        String url = "http://test.com";
        RIFManagerFactory factory = Mockito.mock(RIFManagerFactory.class);
        RIFManager manager = Mockito.mock(RIFManager.class);
        when(factory.getManager(url)).thenReturn(manager);
        // when
        RIFManager result = factory.getManager(url);
        // then
        assertEquals(manager, result);
    }
}

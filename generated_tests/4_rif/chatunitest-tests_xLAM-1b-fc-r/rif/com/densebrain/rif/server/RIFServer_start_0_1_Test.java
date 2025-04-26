package com.densebrain.rif.server;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.net.InetAddress;
import java.rmi.RemoteException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import com.densebrain.rif.server.transport.WebServiceContainer;

@ExtendWith(MockitoExtension.class)
public class RIFServer_start_0_1_Test {

    @Mock
    private WebServiceContainer container;

    @InjectMocks
    private RIFServer rifServer;

    @Test
    public void testStart() throws RemoteException {
        // given
        int port = 8080;
        // when
        rifServer.start();
        // then
        verify(container, times(1)).startContainer();
    }
}

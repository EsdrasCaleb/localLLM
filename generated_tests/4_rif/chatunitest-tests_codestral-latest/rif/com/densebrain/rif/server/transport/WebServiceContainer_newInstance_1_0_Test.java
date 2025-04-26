package com.densebrain.rif.server.transport;

import org.apache.axis2.context.ConfigurationContext;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.net.InetAddress;
import java.rmi.RemoteException;
import java.util.LinkedList;
import java.util.List;
import org.apache.axis2.AxisFault;
import org.apache.axis2.addressing.EndpointReference;
import org.apache.axis2.context.ConfigurationContextFactory;
import org.apache.axis2.description.AxisService;
import org.apache.axis2.rpc.receivers.RPCMessageReceiver;
import org.apache.axis2.transport.http.turnup.SimpleHTTPServer;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

class WebServiceContainer_newInstance_1_0_Test {

    private ConfigurationContext mockConfigurationContext;

    @BeforeEach
    void setUp() {
        mockConfigurationContext = mock(ConfigurationContext.class);
    }

    @Test
    void testNewInstance() throws RemoteException {
        WebServiceContainer container = WebServiceContainer.newInstance(mockConfigurationContext);
        assertNotNull(container);
        assertEquals(mockConfigurationContext, container.configurationContext);
        assertNull(container.contextPath);
        assertNull(container.hostName);
        assertEquals(0, container.port);
        assertNotNull(container.descriptorList);
        assertTrue(container.descriptorList.isEmpty());
    }
}

package com.densebrain.rif.server.transport;

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
import org.apache.axis2.context.ConfigurationContext;
import org.apache.axis2.context.ConfigurationContextFactory;
import org.apache.axis2.description.AxisService;
import org.apache.axis2.rpc.receivers.RPCMessageReceiver;
import org.apache.axis2.transport.http.turnup.SimpleHTTPServer;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public class WebServiceContainer_newInstance_1_0_Test {

    @Test
    public void testNewInstance() throws Exception {
        ConfigurationContext configurationContext = Mockito.mock(ConfigurationContext.class);
        WebServiceContainer container = WebServiceContainer.newInstance(configurationContext);
        assertNotNull(container);
    }

    @Test
    public void testNewInstance_InvalidConfigurationContext() throws Exception {
        ConfigurationContext configurationContext = Mockito.mock(ConfigurationContext.class);
        configurationContext.setProperty("some.property", "some.value");
        WebServiceContainer container = WebServiceContainer.newInstance(configurationContext);
        assertNull(container);
    }

    @Test
    public void testNewInstance_EmptyConfigurationContext() throws Exception {
        WebServiceContainer container = WebServiceContainer.newInstance(null);
        assertNull(container);
    }

    @Test
    public void testNewInstance_NullConfigurationContext() throws Exception {
        assertThrows(NullPointerException.class, () -> WebServiceContainer.newInstance(null));
    }
}
